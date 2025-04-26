package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Field;
import java.io.FileInputStream;
import java.security.KeyStore;
import java.security.PrivateKey;
import java.security.cert.Certificate;
import java.security.cert.X509Certificate;
import java.util.Enumeration;
import org.w3c.dom.Element;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class SimpleSecurityHandlerBean_afterPropertiesSet_1_0_Test {

    @Mock
    private static final Logger logger = LoggerFactory.getLogger(SimpleSecurityHandlerBean.class);

    @InjectMocks
    private SimpleSecurityHandlerBean simpleSecurityHandlerBean;

    @BeforeEach
    public void setUp() {
        // Reset the logger mock before each test
        reset(logger);
    }

    private void setPrivateField(Object target, String fieldName, Object value) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(target, value);
    }

    private Object getPrivateField(Object target, String fieldName) throws Exception {
        Field field = target.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        return field.get(target);
    }

    @Test
    public void testAfterPropertiesSet_BothNull() throws Exception {
        // Arrange
        simpleSecurityHandlerBean.setAlias(null);
        simpleSecurityHandlerBean.setPassword(null);
        // Act
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger, times(1)).warn("Utilzando apelido 'teste' para localizar chave particular no armazém de clientes");
        verify(logger, times(1)).warn("Utilzando senha 'teste' para abrir chave particular no armazém de clientes");
        assertEquals("teste", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("teste".toCharArray(), (char[]) getPrivateField(simpleSecurityHandlerBean, "password"));
    }

    @Test
    public void testAfterPropertiesSet_AliasNull() throws Exception {
        // Arrange
        simpleSecurityHandlerBean.setAlias(null);
        simpleSecurityHandlerBean.setPassword("password".toCharArray());
        // Act
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger, times(1)).warn("Utilzando apelido 'teste' para localizar chave particular no armazém de clientes");
        verify(logger, never()).warn("Utilzando senha 'teste' para abrir chave particular no armazém de clientes");
        assertEquals("teste", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("password".toCharArray(), (char[]) getPrivateField(simpleSecurityHandlerBean, "password"));
    }

    @Test
    public void testAfterPropertiesSet_PasswordNull() throws Exception {
        // Arrange
        simpleSecurityHandlerBean.setAlias("alias");
        simpleSecurityHandlerBean.setPassword(null);
        // Act
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger, never()).warn("Utilzando apelido 'teste' para localizar chave particular no armazém de clientes");
        verify(logger, times(1)).warn("Utilzando senha 'teste' para abrir chave particular no armazém de clientes");
        assertEquals("alias", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("teste".toCharArray(), (char[]) getPrivateField(simpleSecurityHandlerBean, "password"));
    }

    @Test
    public void testAfterPropertiesSet_BothNotNull() throws Exception {
        // Arrange
        simpleSecurityHandlerBean.setAlias("alias");
        simpleSecurityHandlerBean.setPassword("password".toCharArray());
        // Act
        simpleSecurityHandlerBean.afterPropertiesSet();
        // Assert
        verify(logger, never()).warn("Utilzando apelido 'teste' para localizar chave particular no armazém de clientes");
        verify(logger, never()).warn("Utilzando senha 'teste' para abrir chave particular no armazém de clientes");
        assertEquals("alias", getPrivateField(simpleSecurityHandlerBean, "alias"));
        assertArrayEquals("password".toCharArray(), (char[]) getPrivateField(simpleSecurityHandlerBean, "password"));
    }
}
