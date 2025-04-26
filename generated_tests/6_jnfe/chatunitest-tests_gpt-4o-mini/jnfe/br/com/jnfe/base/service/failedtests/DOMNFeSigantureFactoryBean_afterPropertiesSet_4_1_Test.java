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
    public void testAfterPropertiesSet_ProviderNameAndClassNameEmpty() throws Exception {
        System.clearProperty(factoryBean.getProviderName());
        System.clearProperty(factoryBean.getProviderClassName());
        factoryBean.afterPropertiesSet();
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, getPrivateField("providerName"));
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, getPrivateField("providerClassName"));
    }

    @Test
    public void testAfterPropertiesSet_ProviderNameEmpty() throws Exception {
        factoryBean.setProviderClassName("customProviderClass");
        System.clearProperty(factoryBean.getProviderName());
        factoryBean.afterPropertiesSet();
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, getPrivateField("providerName"));
        assertEquals("customProviderClass", getPrivateField("providerClassName"));
    }

    @Test
    public void testAfterPropertiesSet_ProviderClassNameEmpty() throws Exception {
        factoryBean.setProviderName("customProviderName");
        System.clearProperty(factoryBean.getProviderClassName());
        factoryBean.afterPropertiesSet();
        assertEquals("customProviderName", getPrivateField("providerName"));
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, getPrivateField("providerClassName"));
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
