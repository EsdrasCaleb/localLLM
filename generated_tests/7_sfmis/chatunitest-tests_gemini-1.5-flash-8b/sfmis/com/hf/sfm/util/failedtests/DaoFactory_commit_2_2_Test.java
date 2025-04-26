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
public class DaoFactory_commit_2_2_Test {

    @Mock
    private Transaction tx;

    @Mock
    private Connection conn;

    @Mock
    private CallableStatement ps;

    @Mock
    private ResultSet rs;

    private DaoFactory daoFactory;

    @BeforeEach
    void setUp() {
        daoFactory = new DaoFactory();
        // Important: Initialize session
        daoFactory.session = null;
        daoFactory.conn = conn;
        daoFactory.ps = ps;
        daoFactory.rs = rs;
        // Correct access to private field tx
        try {
            java.lang.reflect.Field txField = DaoFactory.class.getDeclaredField("tx");
            txField.setAccessible(true);
            txField.set(daoFactory, tx);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            throw new RuntimeException("Error accessing private field tx", e);
        }
    }

    @Test
    void commit_commitsTransactionAndClosesResources() throws SQLException {
        // Arrange
        doNothing().when(tx).commit();
        doNothing().when(daoFactory).closeAll();
        // Act
        daoFactory.commit();
        // Assert
        verify(tx).commit();
        verify(daoFactory).closeAll();
    }

    @Test
    void commit_throwsSQLException_whenCommitFails() throws SQLException {
        // Arrange
        doThrow(SQLException.class).when(tx).commit();
        // Act & Assert (expecting exception)
        assertThrows(SQLException.class, () -> daoFactory.commit());
        verify(tx).commit();
        verify(daoFactory, never()).closeAll();
    }

    @Test
    void closeAll_isCalled() throws SQLException {
        // Arrange
        doNothing().when(daoFactory).closeAll();
        daoFactory.commit();
        verify(daoFactory).closeAll();
    }

    // Mock for closeAll method (This is crucial for testing closeAll)
    @Test
    void closeAll_handlesNulls() throws SQLException {
        // Arrange
        // Simulate null rs
        daoFactory.rs = null;
        // Simulate null ps
        daoFactory.ps = null;
        // Simulate null conn
        daoFactory.conn = null;
        // Act
        daoFactory.closeAll();
        // Assert - No exceptions should be thrown with nulls
        // No need to verify anything specific here, just that it didn't throw.
    }

    // Correct closeAll method implementation
    private void closeAll() throws SQLException {
        if (rs != null)
            rs.close();
        if (ps != null)
            ps.close();
        if (conn != null)
            conn.close();
    }
}
