package com.hf.sfm.util;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

public class DaoFactory_rollback_4_0_Test {

    @InjectMocks
    private DaoFactory daoFactory;

    @Mock
    private Transaction mockTransaction;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testRollback() {
        // Given
        doNothing().when(mockTransaction).rollback();
        // When
        daoFactory.rollback();
        // Then
        verify(mockTransaction, times(1)).rollback();
    }

    @Test
    public void testRollbackThrowsException() {
        // Given
        doThrow(new RuntimeException("Rollback failed")).when(mockTransaction).rollback();
        // When & Then
        Exception exception = assertThrows(RuntimeException.class, () -> {
            daoFactory.rollback();
        });
        assertEquals("Rollback failed", exception.getMessage());
        verify(mockTransaction, times(1)).rollback();
    }
}
