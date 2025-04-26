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
import com.hf.sfm.crypt.Base64;
import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import /**
 * 此类主要是提供一些常用的方法使用，已在DaoFactoryUtil.java中实例化，业务类只需要继承于DaoFactoryUtil即可调用
 */
com.hf.sfm.crypt.Base64;

@ExtendWith(MockitoExtension.class)
public class DaoFactory_closeSession_1_4_Test {

    @Mock
    private Log log;

    @Mock
    private HibernateSessionFactory hibernateSessionFactory;

    private DaoFactory daoFactory;

    @BeforeEach
    public void setup() {
        MockitoAnnotations.initMocks(this);
        try {
            Field field = DaoFactory.class.getDeclaredField("hibernateSessionFactory");
            field.setAccessible(true);
            field.set(daoFactory, hibernateSessionFactory);
        } catch (NoSuchFieldException | IllegalAccessException e) {
            e.printStackTrace();
        }
    }

    @Test
    public void testCloseSession_ThrowsNullPointerException() {
        // Act and Assert
        assertThrows(NullPointerException.class, () -> daoFactory.closeSession());
    }
}
