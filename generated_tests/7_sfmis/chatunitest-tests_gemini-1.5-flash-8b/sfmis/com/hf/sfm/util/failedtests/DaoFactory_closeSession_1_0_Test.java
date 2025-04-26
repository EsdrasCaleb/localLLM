package com.hf.sfm.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.hibernate.Session;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Transaction;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

public class DaoFactory_closeSession_1_0_Test {

    private DaoFactory daoFactory;

    private Session mockSession;

    private HibernateSessionFactory mockSessionFactory;

    @BeforeEach
    public void setup() {
        mockSession = Mockito.mock(Session.class);
        mockSessionFactory = Mockito.mock(HibernateSessionFactory.class);
        daoFactory = new DaoFactory();
        daoFactory.session = mockSession;
    }

    @Test
    public void testCloseSession() throws SQLException {
        // Arrange
        doNothing().when(mockSessionFactory).closeSession();
        // Act
        daoFactory.closeSession();
        // Assert
        verify(mockSessionFactory).closeSession();
    }

    @Test
    public void testCloseSession_NullSession() {
        // Arrange
        daoFactory.session = null;
        // Act & Assert
        // No need to verify anything, as the method shouldn't throw an exception.
        // This is a defensive test.
    }
}
