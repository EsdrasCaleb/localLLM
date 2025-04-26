package com.hf.sfm.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
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
public class DaoFactory_commit_2_0_Test {

    @Mock
    private Transaction tx;

    @Mock
    private Session session;

    @Mock
    private CallableStatement ps;

    @Mock
    private ResultSet rs;

    @Mock
    private Connection conn;

    @InjectMocks
    private DaoFactory daoFactory;

    @Test
    void testCommit() throws SQLException {
        // Corrected:  No need to manually set these fields. @InjectMocks handles it.
        // daoFactory.session = session;
        // daoFactory.ps = ps;
        // daoFactory.rs = rs;
        // daoFactory.conn = conn;
        // daoFactory.tx = tx;
        // Added to handle potential null pointer exceptions.
        when(session.getTransaction()).thenReturn(tx);
        // Added to handle potential null pointer exceptions.
        when(session.connection()).thenReturn(conn);
        // Act
        daoFactory.commit();
        // Assert
        verify(tx, times(1)).commit();
        verify(session, times(1)).close();
        verify(ps, times(1)).close();
        verify(rs, times(1)).close();
        verify(conn, times(1)).close();
    }

    @Test
    void testCommit_NullTransaction() throws SQLException {
        // Arrange
        // Corrected: No need to manually set tx to null.  @Mock already handles it.
        // daoFactory.tx = null;
        // Simulate null transaction
        when(session.getTransaction()).thenReturn(null);
        // Added to handle potential null pointer exceptions.
        when(session.connection()).thenReturn(conn);
        // Act & Assert
        // Expect no exception when transaction is null.  The behavior is undefined in the production code, but we test for robustness.
        assertDoesNotThrow(() -> daoFactory.commit());
        // Verify that close methods are still called, even if tx is null.
        // Corrected: close should still be called
        verify(session, times(1)).close();
        // Corrected: close should still be called
        verify(ps, times(1)).close();
        // Corrected: close should still be called
        verify(rs, times(1)).close();
        // Corrected: close should still be called
        verify(conn, times(1)).close();
    }
}
