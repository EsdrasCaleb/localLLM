package br.com.jnfe.base.service;

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

@ExtendWith(MockitoExtension.class)
public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_0_Test {

    @Test
    public void testAfterPropertiesSet() throws Exception {
        // Arrange
        DOMNFeSigantureFactoryBean bean = new DOMNFeSigantureFactoryBean();
        // Act
        bean.afterPropertiesSet();
        // Assert
        assertNotNull(bean.getObject());
        assertNotNull(bean.getObjectType());
    }
}
