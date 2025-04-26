package com.hf.sfm.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import org.hibernate.Session;
import org.hibernate.Transaction;
import java.lang.reflect.Method;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.sql.SQLException;
import com.hf.sfm.crypt.Base64;
// Import the class
import com.hf.sfm.util.DaoFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

@ExtendWith(MockitoExtension.class)
class DaoFactory_beginTransaction_3_1_Test {

    @Mock
    private Session mockSession;

    @Mock
    private Transaction mockTransaction;

    @Mock
    private Connection mockConn;

    @Test
    void beginTransaction() throws Exception {
        // Mock the necessary objects
        // Create a DaoFactory instance
        DaoFactory daoFactory = new DaoFactory();
        // Set up the mock session to return the mock transaction
        when(mockSession.beginTransaction()).thenReturn(mockTransaction);
        // Set the session in the DaoFactory instance.  Critically, use the mock
        try {
            Method setSessionMethod = DaoFactory.class.getDeclaredMethod("setCurrentSession", Connection.class);
            setSessionMethod.setAccessible(true);
            setSessionMethod.invoke(daoFactory, mockConn);
        } catch (NoSuchMethodException | IllegalAccessException | java.lang.reflect.InvocationTargetException e) {
            fail("Error setting up the session");
        }
        daoFactory.session = mockSession;
        // Invoke the method under test
        daoFactory.beginTransaction();
        // Verify that beginTransaction was called on the mock session
        verify(mockSession).beginTransaction();
        // Additional assertions (important for comprehensive testing)
        // Check if session is not null
        assertNotNull(daoFactory.session);
        // Access tx field (using reflection is generally discouraged, but necessary here).
        // A better solution would be to make tx a public field or getter method in DaoFactory.
        try {
            Method txMethod = DaoFactory.class.getDeclaredMethod("getTransaction");
            txMethod.setAccessible(true);
            assertNotNull(txMethod.invoke(daoFactory), "Transaction should not be null");
        } catch (NoSuchMethodException | IllegalAccessException | java.lang.reflect.InvocationTargetException e) {
            fail("Error accessing tx field: " + e.getMessage());
        }
    }
}
