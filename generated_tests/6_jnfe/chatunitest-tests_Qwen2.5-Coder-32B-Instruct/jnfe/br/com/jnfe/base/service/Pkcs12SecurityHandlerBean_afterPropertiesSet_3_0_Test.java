package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.FileNotFoundException;
import java.lang.reflect.Field;
import java.security.KeyStore;
import java.security.KeyStoreException;
import java.security.NoSuchAlgorithmException;
import java.security.PrivateKey;
import java.security.UnrecoverableEntryException;
import java.security.cert.X509Certificate;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.KeyStore.PrivateKeyEntry;
import org.w3c.dom.Element;

@ExtendWith(MockitoExtension.class)
public class Pkcs12SecurityHandlerBean_afterPropertiesSet_3_0_Test {

    @Mock
    private Logger logger;

    @InjectMocks
    private Pkcs12SecurityHandlerBean pkcs12SecurityHandlerBean;

    @BeforeEach
    public void setUp() throws Exception {
        // Redirect the logger to the mocked logger
        System.setOut(new java.io.PrintStream(new java.io.OutputStream() {

            public void write(int b) {
                // Redirect System.out to nowhere
            }
        }));
        // Use reflection to set the static final logger field
        Field loggerField = Pkcs12SecurityHandlerBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        Field modifiersField = Field.class.getDeclaredField("modifiers");
        modifiersField.setAccessible(true);
        modifiersField.setInt(loggerField, loggerField.getModifiers() & ~java.lang.reflect.Modifier.FINAL);
        loggerField.set(null, logger);
    }

    @Test
    public void testAfterPropertiesSet_AllValid() throws Exception {
        // Arrange
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        // Act
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger, never()).warn(anyString());
    }

    @Test
    public void testAfterPropertiesSet_PasswordNotSet() throws Exception {
        // Arrange
        pkcs12SecurityHandlerBean.setAlias("testAlias");
        // Act
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger).warn("Senha não inicializada, armazém não será aberto agora.");
    }

    @Test
    public void testAfterPropertiesSet_AliasNotSet() throws Exception {
        // Arrange
        pkcs12SecurityHandlerBean.setPassword("testPassword".toCharArray());
        // Act
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger).warn("Alias (apelido) não inicializado, armazém não será aberto agora.");
    }

    @Test
    public void testAfterPropertiesSet_BothNotSet() throws Exception {
        // Arrange
        // No need to set alias or password
        // Act
        pkcs12SecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger).warn("Alias (apelido) não inicializado, armazém não será aberto agora.");
    }
}
