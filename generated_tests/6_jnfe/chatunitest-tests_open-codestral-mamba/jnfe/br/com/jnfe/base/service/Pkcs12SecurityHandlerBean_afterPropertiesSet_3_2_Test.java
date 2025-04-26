package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.io.FileNotFoundException;
import java.security.KeyStore;
import java.security.KeyStore.PrivateKeyEntry;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_afterPropertiesSet_3_2_Test {

    @Spy
    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        // Arrange
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        // Act
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(pkcs12SecurityHandlerBean, times(1)).loadKeyStore();
    }
}
