package br.com.jnfe.base;

import org.mockito.junit.jupiter.MockitoSettings;
import org.mockito.quality.Strictness;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

@MockitoSettings(strictness = Strictness.LENIENT)
public class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    @Spy
    private DefaultNamespacePrefixMapper defaultNamespacePrefixMapper;

    @InjectMocks
    private DefaultNamespacePrefixMapper defaultNamespacePrefixMapperInjectMocks;

    @Test
    public void testGetPreferredPrefix() {
        Assertions.assertEquals("nf", defaultNamespacePrefixMapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "suggestion", true));
        Assertions.assertNull(defaultNamespacePrefixMapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "suggestion", false));
        Assertions.assertNull(defaultNamespacePrefixMapper.getPreferredPrefix("http://www.portalfiscal.inf.br/other", "suggestion", true));
    }
}
