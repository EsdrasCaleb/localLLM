// Test method
package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.KeyStore;
import javax.net.ssl.KeyManagerFactory;
import org.springframework.beans.factory.InitializingBean;
import br.com.jnfe.base.util.SecurityUtils;

public class TransportKeyStoreBean_toString_7_2_Test {

    @Test
    public void testToString() throws Exception {
        TransportKeyStoreBean transportKeyStoreBean = new TransportKeyStoreBean();
        Field keyStoreUriField = transportKeyStoreBean.getClass().getDeclaredField("keyStoreUri");
        Field keyStoreTypeField = transportKeyStoreBean.getClass().getDeclaredField("keyStoreType");
        Field keyStorePasswordField = transportKeyStoreBean.getClass().getDeclaredField("keyStorePassword");
        Field trustStoreTypeField = transportKeyStoreBean.getClass().getDeclaredField("trustStoreType");
        Field trustStoreField = transportKeyStoreBean.getClass().getDeclaredField("trustStore");
        Field trustStorePasswordField = transportKeyStoreBean.getClass().getDeclaredField("trustStorePassword");
        keyStoreUriField.setAccessible(true);
        keyStoreTypeField.setAccessible(true);
        keyStorePasswordField.setAccessible(true);
        trustStoreTypeField.setAccessible(true);
        trustStoreField.setAccessible(true);
        trustStorePasswordField.setAccessible(true);
        keyStoreUriField.set(transportKeyStoreBean, "file:#{ systemProperties['user.home'] }/jnfe.pfx");
        keyStoreTypeField.set(transportKeyStoreBean, "pkcs12");
        keyStorePasswordField.set(transportKeyStoreBean, "password");
        trustStoreTypeField.set(transportKeyStoreBean, "pkcs12");
        trustStoreField.set(transportKeyStoreBean, "file:#{ systemProperties['user.home'] }/truststore.jks");
        trustStorePasswordField.set(transportKeyStoreBean, "password");
        String expected = "TransportKeyStoreBean@6c2e8e8 [javax.net.ssl.keyStore='file:#{ systemProperties['user.home'] }/jnfe.pfx' " + "javax.net.ssl.keyStoreType='pkcs12' " + "javax.net.ssl.trustStoreType='pkcs12' " + "javax.net.ssl.trustStore='file:#{ systemProperties['user.home'] }/truststore.jks']";
        String actual = transportKeyStoreBean.toString();
        Assertions.assertEquals(expected, actual);
    }
}
