package br.com.jnfe.base;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.slf4j.Logger;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

@ExtendWith(MockitoExtension.class)
@MockitoSettings(strictness = Strictness.LENIENT)
public class TransportKeyStoreBean_afterPropertiesSet_6_2_Test {

    @Mock
    private Logger logger;

    private TransportKeyStoreBean transportKeyStoreBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.initMocks(this);
        transportKeyStoreBean = new TransportKeyStoreBean();
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        transportKeyStoreBean.setKeyStoreUri(null);
        transportKeyStoreBean.setKeyStoreType(null);
        transportKeyStoreBean.setKeyStorePassword(null);
        transportKeyStoreBean.setTrustStore(null);
        transportKeyStoreBean.setTrustStoreType(null);
        transportKeyStoreBean.setTrustStorePassword(null);
        Mockito.doNothing().when(logger).warn(Mockito.anyString());
        Mockito.doNothing().when(logger).info(Mockito.anyString());
        Mockito.doNothing().when(logger).debug(Mockito.anyString());
        transportKeyStoreBean.afterPropertiesSet();
        // Assertions to verify the expected behavior
    }
}
