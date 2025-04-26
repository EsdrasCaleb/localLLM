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
    public void testGetPreferredPrefix() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String expectedPrefix = "nf";
        String actualPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "nf", true);
        assertTrue(actualPrefix == expectedPrefix);
        expectedPrefix = "nf";
        actualPrefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "nf", false);
        assertFalse(actualPrefix == expectedPrefix);
    }
}
