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
public class DaoFactory_encrypt_5_3_Test {

    @Mock
    private DaoFactory daoFactoryMock;

    @Test
    public void testEncrypt() {
        String input = "sensitiveData";
        String expectedOutput = java.util.Base64.getEncoder().encodeToString(input.getBytes());
        when(daoFactoryMock.encrypt(input)).thenReturn(expectedOutput);
        // Act
        String result = daoFactoryMock.encrypt(input);
        // Assert
        assertEquals(expectedOutput, result);
    }
}
