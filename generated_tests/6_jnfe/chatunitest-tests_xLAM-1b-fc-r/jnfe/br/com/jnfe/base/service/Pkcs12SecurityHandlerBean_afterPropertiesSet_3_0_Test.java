package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
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
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_afterPropertiesSet_3_0_Test {

    @Mock
    private Logger logger;

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @Test
    public void afterPropertiesSet_aliasNotNull_shouldCallLoadKeyStore() throws Exception {
        // given
        String alias = "alias";
        char[] password = { 'p', 'a', 's', 's', 'w', 'o', 'r', 'd' };
        pkcs12SecurityHandlerBean.setAlias(alias);
        pkcs12SecurityHandlerBean.setPassword(password);
        // when
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // then
        verify(logger).warn("Senha n�o inicializada, armaz�m n�o ser� aberto agora.");
    }

    @Test
    public void afterPropertiesSet_passwordNotNull_shouldNotCallLoadKeyStore() throws Exception {
        // given
        String alias = "alias";
        char[] password = { 'p', 'a', 's', 's', 'w', 'o', 'r', 'd' };
        pkcs12SecurityHandlerBean.setAlias(alias);
        pkcs12SecurityHandlerBean.setPassword(password);
        // when
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // then
        verify(logger, never()).warn("Senha n�o inicializada, armaz�m n�o ser� aberto agora.");
    }
}
