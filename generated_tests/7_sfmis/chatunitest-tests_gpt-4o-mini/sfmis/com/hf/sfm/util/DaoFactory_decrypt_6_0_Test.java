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

public class DaoFactory_decrypt_6_0_Test {

    private DaoFactory daoFactory;

    @BeforeEach
    public void setUp() {
        daoFactory = new DaoFactory();
    }

    @Test
    public void testDecrypt_ValidBase64String() {
        // "Hello World" in Base64
        String input = "SGVsbG8gV29ybGQ=";
        String expected = "Hello World";
        String result = daoFactory.decrypt(input);
        assertEquals(expected, result);
    }

    @Test
    public void testDecrypt_EmptyString() {
        // Empty Base64 string
        String input = "";
        String expected = "";
        String result = daoFactory.decrypt(input);
        assertEquals(expected, result);
    }

    @Test
    public void testDecrypt_InvalidBase64String() {
        // Invalid Base64 string
        String input = "!!!InvalidBase64!!!";
        assertThrows(IllegalArgumentException.class, () -> {
            daoFactory.decrypt(input);
        });
    }
}
