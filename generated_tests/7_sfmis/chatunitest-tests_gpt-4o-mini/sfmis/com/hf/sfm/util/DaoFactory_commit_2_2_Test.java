package com.hf.sfm.util;

import org.hibernate.Transaction;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.sql.CallableStatement;
import java.sql.Connection;
import java.sql.ResultSet;
import java.sql.SQLException;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import org.hibernate.Session;
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

    @InjectMocks
    private DaoFactory daoFactory;

    @Mock
    private Transaction mockTransaction;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        // Using reflection to set the private field tx
        try {
            java.lang.reflect.Field txField = DaoFactory.class.getDeclaredField("tx");
            txField.setAccessible(true);
            txField.set(daoFactory, mockTransaction);
        } catch (Exception e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }

    @Test
    public void testCommit() {
        // Arrange
        doNothing().when(mockTransaction).commit();
        // Act
        daoFactory.commit();
        // Assert
        verify(mockTransaction).commit();
        // Verify that closeAll() was called, using reflection
        try {
            java.lang.reflect.Method closeAllMethod = DaoFactory.class.getDeclaredMethod("closeAll");
            closeAllMethod.setAccessible(true);
            // Verify that closeAll() was called
            verify(closeAllMethod).invoke(daoFactory);
        } catch (Exception e) {
            fail("Reflection failed: " + e.getMessage());
        }
    }
}
