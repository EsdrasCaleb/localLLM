package com.hf.sfm.util;

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
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Transaction;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

public class DaoFactory_closeAll_9_0_Test {

    private DaoFactory daoFactory;

    @Mock
    private ResultSet rs;

    @Mock
    private CallableStatement ps;

    @Mock
    private Connection conn;

    @Mock
    private Session session;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        daoFactory = new DaoFactory();
        daoFactory.rs = rs;
        daoFactory.ps = ps;
        daoFactory.conn = conn;
        daoFactory.session = session;
    }

    @Test
    public void testCloseAll_WithNullResources() throws SQLException {
        // Arrange
        daoFactory.rs = null;
        daoFactory.ps = null;
        daoFactory.conn = null;
        daoFactory.session = null;
        // Act
        daoFactory.closeAll();
        // Assert
        // No exceptions should be thrown and no methods should be called
        verify(rs, never()).close();
        verify(ps, never()).close();
        verify(conn, never()).close();
        verify(session, never()).close();
    }
}
