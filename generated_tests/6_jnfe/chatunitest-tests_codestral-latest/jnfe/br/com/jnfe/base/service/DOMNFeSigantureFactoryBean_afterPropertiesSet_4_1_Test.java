package br.com.jnfe.base.service;

import java.lang.reflect.Field;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_1_Test {

    @InjectMocks
    private DOMNFeSigantureFactoryBean domNFeSigantureFactoryBean;

    private static final Logger logger = LoggerFactory.getLogger(DOMNFeSigantureFactoryBean_afterPropertiesSet_4_1_Test.class);

    @BeforeEach
    void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    void testAfterPropertiesSet_DefaultValues() throws Exception {
        // Arrange
        setField(domNFeSigantureFactoryBean, "providerName", "");
        setField(domNFeSigantureFactoryBean, "providerClassName", "");
        // Act
        domNFeSigantureFactoryBean.afterPropertiesSet();
        // Assert
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, domNFeSigantureFactoryBean.getProviderName());
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, domNFeSigantureFactoryBean.getProviderClassName());
    }

    @Test
    void testAfterPropertiesSet_CustomValues() throws Exception {
        // Arrange
        setField(domNFeSigantureFactoryBean, "providerName", "customProviderName");
        setField(domNFeSigantureFactoryBean, "providerClassName", "customProviderClassName");
        // Act
        domNFeSigantureFactoryBean.afterPropertiesSet();
        // Assert
        assertEquals("customProviderName", domNFeSigantureFactoryBean.getProviderName());
        assertEquals("customProviderClassName", domNFeSigantureFactoryBean.getProviderClassName());
    }

    @Test
    void testAfterPropertiesSet_SystemProperty() throws Exception {
        // Arrange
        setField(domNFeSigantureFactoryBean, "providerName", "customProviderName");
        setField(domNFeSigantureFactoryBean, "providerClassName", "customProviderClassName");
        System.setProperty("customProviderName", "systemPropertyValue");
        // Act
        domNFeSigantureFactoryBean.afterPropertiesSet();
        // Assert
        assertEquals("systemPropertyValue", domNFeSigantureFactoryBean.getProviderName());
        assertEquals("customProviderClassName", domNFeSigantureFactoryBean.getProviderClassName());
        // Clean up
        System.clearProperty("customProviderName");
    }

    private void setField(Object object, String fieldName, Object fieldValue) throws Exception {
        Field field = object.getClass().getDeclaredField(fieldName);
        field.setAccessible(true);
        field.set(object, fieldValue);
    }
}
