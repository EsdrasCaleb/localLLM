package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class TransportKeyStoreBean_afterPropertiesSet_6_0_Test {

    private static final Logger logger = LoggerFactory.getLogger(TransportKeyStoreBean_afterPropertiesSet_6_0_Test.class);

    @Test
    void testAfterPropertiesSet_allPropertiesSet() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.setKeyStoreUri("testUri");
        bean.setKeyStoreType("testType");
        bean.setKeyStorePassword("testPassword");
        bean.setTrustStore("testTrustStore");
        bean.setTrustStoreType("testTrustStoreType");
        bean.setTrustStorePassword("testTrustStorePassword");
        bean.afterPropertiesSet();
        assertEquals("testType", System.getProperty("javax.net.ssl.keyStoreType"));
        assertEquals("testUri", System.getProperty("javax.net.ssl.keyStore"));
        assertEquals("testPassword", System.getProperty("javax.net.ssl.keyStorePassword"));
        assertEquals("testTrustStore", System.getProperty("javax.net.ssl.trustStore"));
        assertEquals("testTrustStoreType", System.getProperty("javax.net.ssl.trustStoreType"));
        assertEquals("testTrustStorePassword", System.getProperty("javax.net.ssl.trustStorePassword"));
    }

    @Test
    void testAfterPropertiesSet_defaultKeyStoreType() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        Field defaultKeyStoreTypeField = TransportKeyStoreBean.class.getDeclaredField("DEFAULT_KEYSTORE_TYPE");
        defaultKeyStoreTypeField.setAccessible(true);
        String defaultKeyStoreType = (String) defaultKeyStoreTypeField.get(bean);
        assertEquals(defaultKeyStoreType, System.getProperty("javax.net.ssl.keyStoreType"));
    }

    @Test
    void testAfterPropertiesSet_defaultKeyStoreUri() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        Field defaultKeyStoreUriField = TransportKeyStoreBean.class.getDeclaredField("DEFAULT_KEYSTORE_URI");
        defaultKeyStoreUriField.setAccessible(true);
        String defaultKeyStoreUri = (String) defaultKeyStoreUriField.get(bean);
        assertEquals(defaultKeyStoreUri, System.getProperty("javax.net.ssl.keyStore"));
    }

    @Test
    void testAfterPropertiesSet_nullKeyStorePassword() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        assertNull(System.getProperty("javax.net.ssl.keyStorePassword"));
    }

    @Test
    void testAfterPropertiesSet_emptyKeyStorePassword() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.setKeyStorePassword("");
        bean.afterPropertiesSet();
        assertNull(System.getProperty("javax.net.ssl.keyStorePassword"));
    }

    @Test
    void testAfterPropertiesSet_nullTrustStore() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        assertNull(System.getProperty("javax.net.ssl.trustStore"));
    }

    @Test
    void testAfterPropertiesSet_nullTrustStoreType() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        assertNull(System.getProperty("javax.net.ssl.trustStoreType"));
    }

    @Test
    void testAfterPropertiesSet_nullTrustStorePassword() throws Exception {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.afterPropertiesSet();
        assertNull(System.getProperty("javax.net.ssl.trustStorePassword"));
    }
}
