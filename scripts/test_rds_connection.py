"""
Test de conexión a AWS RDS.
Verifica que podemos conectar a la base de datos textil.
"""

import mysql.connector
from mysql.connector import Error
import sys
from pathlib import Path

# Añadir parent al path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings


def test_rds_connection():
    """Prueba conexión a RDS y muestra información de la BD"""

    print("=" * 70)
    print("TESTING AWS RDS CONNECTION")
    print("=" * 70)

    print(f"\n📍 Connection Details:")
    print(f"   Host: {settings.MYSQL_HOST}")
    print(f"   Port: {settings.MYSQL_PORT}")
    print(f"   User: {settings.MYSQL_USER}")
    print(f"   Database: {settings.MYSQL_DATABASE}")

    try:
        print("\n🔌 Connecting to RDS...")

        connection = mysql.connector.connect(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            database=settings.MYSQL_DATABASE,
            connect_timeout=10
        )

        if connection.is_connected():
            db_info = connection.get_server_info()
            print(f"✅ Connected to MySQL Server version {db_info}")

            cursor = connection.cursor()

            # Test 1: Ver bases de datos disponibles
            cursor.execute("SHOW DATABASES;")
            databases = cursor.fetchall()
            print(f"\n📁 Available Databases ({len(databases)}):")
            for db in databases:
                marker = "👉" if db[0] == settings.MYSQL_DATABASE else "  "
                print(f"   {marker} {db[0]}")

            # Test 2: Ver tablas en la BD textil
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print(f"\n📊 Tables in '{settings.MYSQL_DATABASE}' ({len(tables)}):")

            if not tables:
                print("   ⚠️  No tables found in database!")
                print("   This database might be empty.")
            else:
                for table in tables:
                    print(f"   - {table[0]}")

                # Test 3: Contar registros en cada tabla
                print(f"\n📈 Row Counts:")
                total_rows = 0
                for table in tables:
                    table_name = table[0]
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`;")
                        count = cursor.fetchone()[0]
                        total_rows += count
                        print(f"   - {table_name:30s}: {count:>10,} rows")
                    except Error as e:
                        print(f"   - {table_name:30s}: ERROR ({e})")

                print(f"   {'TOTAL':30s}: {total_rows:>10,} rows")

            # Test 4: Verificar permisos del usuario
            print(f"\n🔐 User Permissions:")
            cursor.execute("SHOW GRANTS FOR CURRENT_USER();")
            grants = cursor.fetchall()
            for grant in grants:
                # Limpiar el output para que sea más legible
                grant_str = grant[0]
                if 'ALL PRIVILEGES' in grant_str:
                    print(f"   ✅ {grant_str}")
                elif 'SELECT' in grant_str:
                    print(f"   ✅ {grant_str}")
                elif 'INSERT' in grant_str or 'UPDATE' in grant_str or 'DELETE' in grant_str:
                    print(f"   ⚠️  {grant_str}")
                    print(f"      WARNING: User has write permissions!")
                else:
                    print(f"   ℹ️  {grant_str}")

            # Test 5: Probar una query SELECT simple
            print(f"\n🔍 Testing SELECT Query:")
            if tables:
                first_table = tables[0][0]
                cursor.execute(f"SELECT * FROM `{first_table}` LIMIT 3;")
                results = cursor.fetchall()

                if results:
                    print(f"   ✅ Successfully queried '{first_table}'")
                    print(f"   Sample: Retrieved {len(results)} row(s)")

                    # Mostrar columnas
                    columns = [desc[0] for desc in cursor.description]
                    print(f"   Columns: {', '.join(columns)}")
                else:
                    print(f"   ⚠️  Table '{first_table}' is empty")

            cursor.close()
            connection.close()

            print("\n" + "=" * 70)
            print("✅ CONNECTION TEST PASSED")
            print("=" * 70)
            print("\n💡 Next steps:")
            print("   1. Review the table structure above")
            print("   2. Update data/sql_examples.json with relevant examples")
            print("   3. Proceed to PASO 2: docker-compose.yml")
            print("=" * 70)

            return True

    except Error as e:
        print(f"\n❌ ERROR: {e}")
        print("\n" + "=" * 70)
        print("🔧 TROUBLESHOOTING GUIDE")
        print("=" * 70)

        if "Access denied" in str(e):
            print("\n❌ Authentication Error:")
            print("   - Verify username/password in .env are correct")
            print("   - Check if user exists in RDS")

        elif "Can't connect" in str(e) or "timed out" in str(e):
            print("\n❌ Connection Error:")
            print("   1. Check RDS Security Group:")
            print("      AWS Console → RDS → textil → Connectivity & security")
            print("      → VPC security groups → Inbound rules")
            print("      → Must allow MySQL/Aurora (port 3306) from your IP")
            print("")
            print("   2. Check if RDS is publicly accessible:")
            print("      AWS Console → RDS → textil → Connectivity & security")
            print("      → Publicly accessible: should be 'Yes' for local testing")
            print("")
            print("   3. Verify your IP address:")
            print("      curl ifconfig.me")
            print("      Add this IP to Security Group")

        elif "Unknown database" in str(e):
            print("\n❌ Database Not Found:")
            print("   - Verify MYSQL_DATABASE='textil' is correct")
            print("   - Check available databases after connecting")

        else:
            print("\n❌ Unknown Error:")
            print("   - Check RDS status in AWS Console")
            print("   - Verify RDS endpoint is correct")
            print("   - Try connecting with MySQL Workbench for more details")

        print("\n" + "=" * 70)

        return False


if __name__ == "__main__":
    try:
        success = test_rds_connection()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)