package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
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
import org.slf4j.LoggerFactory;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_afterPropertiesSet_3_0_Test {

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @Mock
    private Logger logger;

    @BeforeEach
    public void setUp() {
        pkcs12SecurityHandlerBean = new Pkcs12SecurityHandlerBean();
    }

    @Test
    public void testAfterPropertiesSet_ValidAliasAndPassword() throws Exception {
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        verify(logger, never()).warn(anyString());
    }

    @Test
    public void testAfterPropertiesSet_NullAlias() throws Exception {
        pkcs12SecurityHandlerBean.setAlias(null);
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        verify(logger).warn("Alias (apelido) n�o inicializado, armaz�m n�o ser� aberto agora.");
    }

    @Test
    public void testAfterPropertiesSet_EmptyAlias() throws Exception {
        pkcs12SecurityHandlerBean.setAlias("");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        verify(logger).warn("Alias (apelido) n�o inicializado, armaz�m n�o ser� aberto agora.");
    }

    @Test
    public void testAfterPropertiesSet_NullPassword() throws Exception {
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword(null);
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        verify(logger).warn("Senha n�o inicializada, armaz�m n�o ser� aberto agora.");
    }

    @Test
    public void testAfterPropertiesSet_EmptyPassword() throws Exception {
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword(new char[0]);
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        verify(logger).warn("Senha n�o inicializada, armaz�m n�o ser� aberto agora.");
    }
}
