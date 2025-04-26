package br.com.jnfe.base.service;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;

public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_1_Test {

    private DOMNFeSigantureFactoryBean factoryBean;

    @BeforeEach
    public void setUp() {
        factoryBean = new DOMNFeSigantureFactoryBean();
    }

    @Test
    public void testAfterPropertiesSet_WithSystemProperty() throws Exception {
        factoryBean.setProviderName("customProviderName");
        System.setProperty("customProviderName", "systemProviderClass");
        factoryBean.afterPropertiesSet();
        assertEquals("systemProviderClass", getPrivateField("providerName"));
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, getPrivateField("providerClassName"));
    }

    private String getPrivateField(String fieldName) throws Exception {
        Field field = DOMNFeSigantureFactoryBean.class.getDeclaredField(fieldName);
        field.setAccessible(true);
        return (String) field.get(factoryBean);
    }
}
