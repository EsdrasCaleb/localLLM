package br.com.jnfe.base.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.lang.reflect.Field;
import java.util.Properties;
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
class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_3_Test {

    private static final Logger logger = LoggerFactory.getLogger(DOMNFeSigantureFactoryBean_afterPropertiesSet_4_3_Test.class);

    @Test
    void afterPropertiesSet_bothEmpty_setsDefaultsAndLogsWarning() throws Exception {
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        bean.afterPropertiesSet();
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_NAME, bean.getProviderName());
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
        // Verify logs - requires mocking LoggerFactory if you want precise log verification.  This example just checks for the presence of warnings.
        // More sophisticated logging verification would be needed for production code.
        Logger mockLogger = mock(Logger.class);
        Field loggerField = DOMNFeSigantureFactoryBean.class.getDeclaredField("logger");
        loggerField.setAccessible(true);
        loggerField.set(bean, mockLogger);
        verify(mockLogger, atLeastOnce()).warn(anyString(), anyString());
    }

    @Test
    void afterPropertiesSet_providerNameSet_usesProviderName() throws Exception {
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        bean.setProviderName("myProvider");
        bean.afterPropertiesSet();
        assertEquals("myProvider", bean.getProviderName());
        assertEquals(DOMNFeSigantureFactoryBean.DEFAULT_PROVIDER_CLASS_NAME, bean.getProviderClassName());
    }

    @Test
    void afterPropertiesSet_providerClassNameSet_usesSystemPropertyIfAvailable() throws Exception {
        String systemPropValue = "systemProvider";
        System.setProperty("myProvider", systemPropValue);
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        bean.setProviderName("myProvider");
        bean.setProviderClassName("someClass");
        bean.afterPropertiesSet();
        assertEquals(systemPropValue, bean.getProviderName());
        assertEquals("someClass", bean.getProviderClassName());
        // Clean up System property
        System.clearProperty("myProvider");
    }

    @Test
    void afterPropertiesSet_providerClassNameSet_usesClassNameIfSystemPropertyNotAvailable() throws Exception {
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        bean.setProviderName("myProvider");
        bean.setProviderClassName("someClass");
        bean.afterPropertiesSet();
        assertEquals("someClass", bean.getProviderName());
        assertEquals("someClass", bean.getProviderClassName());
    }

    @Test
    void afterPropertiesSet_bothSet_usesProviderName() throws Exception {
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        bean.setProviderName("myProvider");
        bean.setProviderClassName("someClass");
        bean.afterPropertiesSet();
        assertEquals("myProvider", bean.getProviderName());
        assertEquals("someClass", bean.getProviderClassName());
    }
}
