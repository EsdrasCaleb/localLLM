package com.hf.sfm.util;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
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

@ExtendWith(MockitoExtension.class)
public class DaoFactory_rollback_4_0_Test {

    @Mock
    private Transaction mockTransaction;

    @InjectMocks
    private DaoFactory daoFactory;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testRollback() {
        // Arrange
        doNothing().when(mockTransaction).rollback();
        // Act
        daoFactory.rollback();
        // Assert
        verify(mockTransaction, times(1)).rollback();
    }
}
