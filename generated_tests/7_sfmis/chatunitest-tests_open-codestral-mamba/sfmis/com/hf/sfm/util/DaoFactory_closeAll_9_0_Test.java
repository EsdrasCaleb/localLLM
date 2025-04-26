package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Session;
import org.hibernate.Transaction;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

@ExtendWith(MockitoExtension.class)
public class DaoFactory_closeAll_9_0_Test {

    @Mock
    private Session mockSession;

    @Mock
    private CallableStatement mockPs;

    @Mock
    private ResultSet mockRs;

    @Mock
    private Connection mockConn;

    @InjectMocks
    private DaoFactory daoFactory;

    @Test
    public void testCloseAll() throws SQLException {
        // Arrange
        daoFactory.session = mockSession;
        daoFactory.rs = mockRs;
        daoFactory.ps = mockPs;
        daoFactory.conn = mockConn;
        // Act
        daoFactory.closeAll();
        // Assert
        // <Buggy Line>: unreported exception java.sql.SQLException; must be caught or declared to be thrown
        verify(mockRs, times(1)).close();
        // <Buggy Line>: unreported exception java.sql.SQLException; must be caught or declared to be thrown
        verify(mockPs, times(1)).close();
        verify(mockConn, times(1)).close();
        verify(mockSession, times(1)).close();
    }
}
