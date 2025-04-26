package br.com.jnfe.base;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

public class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    @Test
    public void testGetPreferredPrefix_NFe() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String preferredPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "nf", true);
        assertEquals("nf", preferredPrefix);
    }

    @Test
    public void testGetPreferredPrefix_NoNFe() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String preferredPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "nf", false);
        assertNotNull(preferredPrefix);
    }

    @Test
    public void testGetPreferredPrefix_NoNFe_NoSuggestion() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String preferredPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", null, false);
        assertNotNull(preferredPrefix);
    }

    @Test
    public void testGetPreferredPrefix_NoNFe_NoSuggestion_NoRequirePrefix() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String preferredPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", null, false);
        assertNotNull(preferredPrefix);
    }
}
