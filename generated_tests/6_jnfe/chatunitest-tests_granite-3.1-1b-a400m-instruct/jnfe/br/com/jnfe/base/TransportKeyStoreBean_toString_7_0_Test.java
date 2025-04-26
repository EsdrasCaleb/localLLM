package br.com.jnfe.base;

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

public class TransportKeyStoreBean_toString_7_0_Test {

    @Test
    void testToString() {
        TransportKeyStoreBean bean = new TransportKeyStoreBean();
        bean.setKeyStoreUri("file://");
        bean.setKeyStoreType("pkcs12");
        bean.setKeyStorePassword("password");
        bean.setTrustStore("truststore.pfx");
        bean.setTrustStoreType("pkcs12");
        bean.setTrustStorePassword("trustpassword");
        String expectedString = "TransportKeyStoreBean@12345678 [file://]";
        String actualString = bean.toString();
        assertEquals(expectedString, actualString);
    }
}
