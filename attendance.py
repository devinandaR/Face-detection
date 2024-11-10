import sqlite3

def delete_attendance_logs():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    cursor.execute('''
        DELETE FROM attendance_logs
    ''')
    conn.commit()
    conn.close()
    return "Attendance logs deleted successfully"

if __name__ == "__main__":
    print(delete_attendance_logs())
