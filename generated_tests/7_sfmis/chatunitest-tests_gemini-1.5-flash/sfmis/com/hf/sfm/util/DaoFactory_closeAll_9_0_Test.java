package com.hf.sfm.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.hibernate.Session;
import org.hibernate.Transaction;
import com.hf.sfm.crypt.Base64;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

@ExtendWith(MockitoExtension.class)
class DaoFactory_closeAll_9_0_Test {

    @Test
    void testCloseAll_allNotNull() throws SQLException {
        // Arrange
        DaoFactory daoFactory = new DaoFactory();
        ResultSet mockResultSet = Mockito.mock(ResultSet.class);
        CallableStatement mockCallableStatement = Mockito.mock(CallableStatement.class);
        Connection mockConnection = Mockito.mock(Connection.class);
        Session mockSession = Mockito.mock(Session.class);
        daoFactory.rs = mockResultSet;
        daoFactory.ps = mockCallableStatement;
        daoFactory.conn = mockConnection;
        daoFactory.session = mockSession;
        // Act
        daoFactory.closeAll();
        // Assert
        verify(mockResultSet, times(1)).close();
        verify(mockCallableStatement, times(1)).close();
        verify(mockConnection, times(1)).close();
        verify(mockSession, times(1)).close();
    }

    @Test
    void testCloseAll_rsNull() throws SQLException {
        DaoFactory daoFactory = new DaoFactory();
        CallableStatement mockCallableStatement = Mockito.mock(CallableStatement.class);
        Connection mockConnection = Mockito.mock(Connection.class);
        Session mockSession = Mockito.mock(Session.class);
        daoFactory.ps = mockCallableStatement;
        daoFactory.conn = mockConnection;
        daoFactory.session = mockSession;
        daoFactory.closeAll();
        verify(mockCallableStatement, times(1)).close();
        verify(mockConnection, times(1)).close();
        verify(mockSession, times(1)).close();
    }

    @Test
    void testCloseAll_psNull() throws SQLException {
        DaoFactory daoFactory = new DaoFactory();
        ResultSet mockResultSet = Mockito.mock(ResultSet.class);
        Connection mockConnection = Mockito.mock(Connection.class);
        Session mockSession = Mockito.mock(Session.class);
        daoFactory.rs = mockResultSet;
        daoFactory.conn = mockConnection;
        daoFactory.session = mockSession;
        daoFactory.closeAll();
        verify(mockResultSet, times(1)).close();
        verify(mockConnection, times(1)).close();
        verify(mockSession, times(1)).close();
    }

    @Test
    void testCloseAll_connNull() throws SQLException {
        DaoFactory daoFactory = new DaoFactory();
        ResultSet mockResultSet = Mockito.mock(ResultSet.class);
        CallableStatement mockCallableStatement = Mockito.mock(CallableStatement.class);
        Session mockSession = Mockito.mock(Session.class);
        daoFactory.rs = mockResultSet;
        daoFactory.ps = mockCallableStatement;
        daoFactory.session = mockSession;
        daoFactory.closeAll();
        verify(mockResultSet, times(1)).close();
        verify(mockCallableStatement, times(1)).close();
        verify(mockSession, times(1)).close();
    }

    @Test
    void testCloseAll_sessionNull() throws SQLException {
        DaoFactory daoFactory = new DaoFactory();
        ResultSet mockResultSet = Mockito.mock(ResultSet.class);
        CallableStatement mockCallableStatement = Mockito.mock(CallableStatement.class);
        Connection mockConnection = Mockito.mock(Connection.class);
        daoFactory.rs = mockResultSet;
        daoFactory.ps = mockCallableStatement;
        daoFactory.conn = mockConnection;
        daoFactory.closeAll();
        verify(mockResultSet, times(1)).close();
        verify(mockCallableStatement, times(1)).close();
        verify(mockConnection, times(1)).close();
    }

    @Test
    void testCloseAll_allNull() {
    }
}
