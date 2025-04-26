package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.security.KeyStore;
import java.security.NoSuchAlgorithmException;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileNotFoundException;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.PrivateKey;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_afterPropertiesSet_3_1_Test {

    @Mock
    private Logger logger;

    @InjectMocks
    private Pkcs12SecurityHandlerBean bean;

    @Test
    void afterPropertiesSet_validAliasAndPassword_loadsKeyStore() throws Exception {
        bean.setAlias("myAlias");
        bean.setPassword("myPassword".toCharArray());
        // We cannot directly test loadKeyStore() as it's private.  We'll use reflection to verify it's called.
        Method loadKeyStoreMethod = Pkcs12SecurityHandlerBean.class.getDeclaredMethod("loadKeyStore");
        loadKeyStoreMethod.setAccessible(true);
        loadKeyStoreMethod.invoke(bean);
        verify(logger, never()).warn(anyString());
    }

    @Test
    void afterPropertiesSet_nullAlias_logsWarning() throws Exception {
        bean.setAlias(null);
        bean.setPassword("myPassword".toCharArray());
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
    }

    @Test
    void afterPropertiesSet_emptyAlias_logsWarning() throws Exception {
        bean.setAlias("");
        bean.setPassword("myPassword".toCharArray());
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
    }

    @Test
    void afterPropertiesSet_validAlias_nullPassword_logsWarning() throws Exception {
        bean.setAlias("myAlias");
        bean.setPassword(null);
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
    }

    @Test
    void afterPropertiesSet_validAlias_emptyPassword_logsWarning() throws Exception {
        bean.setAlias("myAlias");
        bean.setPassword("".toCharArray());
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
    }

    @Test
    void afterPropertiesSet_nullAliasAndPassword_logsWarning() throws Exception {
        bean.setAlias(null);
        bean.setPassword(null);
        bean.afterPropertiesSet();
        verify(logger, times(1)).warn(anyString());
    }

    @Test
    void getCertificate_success() throws NoSuchAlgorithmException, UnrecoverableEntryException, KeyStoreException {
        // Mocking the necessary parts for a successful certificate retrieval.
        KeyStore keyStore = mock(KeyStore.class);
        X509Certificate cert = mock(X509Certificate.class);
        when(keyStore.getCertificate("myAlias")).thenReturn(cert);
        // Using reflection to set the keystore in the private field.
        try {
            Method setKeyStoreMethod = Pkcs12SecurityHandlerBean.class.getDeclaredMethod("setKeyStore", KeyStore.class);
            setKeyStoreMethod.setAccessible(true);
            setKeyStoreMethod.invoke(bean, keyStore);
        } catch (NoSuchMethodException | IllegalAccessException | InvocationTargetException e) {
            fail("Failed to set KeyStore using reflection: " + e.getMessage());
        }
        bean.setAlias("myAlias");
        bean.setPassword("myPassword".toCharArray());
        X509Certificate result = bean.getCertificate();
        assertNotNull(result);
        assertEquals(cert, result);
    }
}
