package br.com.jnfe.base.service;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.security.Provider;
import javax.xml.crypto.dsig.XMLSignatureFactory;
import org.springframework.beans.factory.FactoryBean;
import org.springframework.beans.factory.InitializingBean;

@ExtendWith(MockitoExtension.class)
public class DOMNFeSigantureFactoryBean_afterPropertiesSet_4_2_Test {

    @Spy
    @InjectMocks
    private DOMNFeSigantureFactoryBean domNFeSigantureFactoryBean;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
    }

    @Test
    public void testAfterPropertiesSet() throws Exception {
        doReturn("customProviderName").when(domNFeSigantureFactoryBean).getProviderName();
        doReturn("customProviderClassName").when(domNFeSigantureFactoryBean).getProviderClassName();
        domNFeSigantureFactoryBean.afterPropertiesSet();
        assertEquals("customProviderName", System.getProperty("customProviderName"));
        assertEquals("customProviderClassName", domNFeSigantureFactoryBean.getProviderClassName());
    }
}
