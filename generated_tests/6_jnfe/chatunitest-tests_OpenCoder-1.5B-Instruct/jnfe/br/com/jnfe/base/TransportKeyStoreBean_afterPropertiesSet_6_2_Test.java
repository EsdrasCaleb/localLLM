package br.com.jnfe.base;

import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

public class TransportKeyStoreBean_afterPropertiesSet_6_2_Test {

    private TransportKeyStoreBean transportKeyStoreBean;

    private Method afterPropertiesSet;

    @BeforeEach
    public void setUp() throws Exception {
        transportKeyStoreBean = Mockito.mock(TransportKeyStoreBean.class);
        afterPropertiesSet = transportKeyStoreBean.getClass().getDeclaredMethod("afterPropertiesSet");
        afterPropertiesSet.setAccessible(true);
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        // Arrange
        transportKeyStoreBean.setKeyStoreUri("file:#{ systemProperties['user.home'] }/jnfe.pfx");
        transportKeyStoreBean.setKeyStoreType("pkcs12");
        transportKeyStoreBean.setKeyStorePassword("securePassword");
        transportKeyStoreBean.setTrustStore("trustStorePath");
        transportKeyStoreBean.setTrustStoreType("JKS");
        transportKeyStoreBean.setTrustStorePassword("trustPassword");
        // Act
        afterPropertiesSet.invoke(transportKeyStoreBean);
    }
}
