package com.hf.sfm.util;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Session;
import org.hibernate.Transaction;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Field;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

@ExtendWith(MockitoExtension.class)
public class DaoFactory_beginTransaction_3_0_Test {

    @Mock
    private Session mockSession;

    @InjectMocks
    private DaoFactory daoFactory;

    @Test
    void testBeginTransaction() throws NoSuchFieldException, IllegalAccessException {
        // Mock the session's beginTransaction() method to return a mock Transaction
        Transaction mockTransaction = mock(Transaction.class);
        when(mockSession.beginTransaction()).thenReturn(mockTransaction);
        // Invoke the method under test
        daoFactory.beginTransaction();
        // Verify that the session's beginTransaction() method was called once
        verify(mockSession, times(1)).beginTransaction();
        // Access the private field using reflection
        Field txField = DaoFactory.class.getDeclaredField("tx");
        txField.setAccessible(true);
        Transaction actualTransaction = (Transaction) txField.get(daoFactory);
        assertEquals(mockTransaction, actualTransaction);
    }

    @Test
    void testBeginTransaction_NullSession() throws NoSuchFieldException, IllegalAccessException {
        // Set session to null to test that branch
        Field sessionField = DaoFactory.class.getDeclaredField("session");
        sessionField.setAccessible(true);
        sessionField.set(daoFactory, null);
        assertThrows(NullPointerException.class, () -> daoFactory.beginTransaction());
    }
}
