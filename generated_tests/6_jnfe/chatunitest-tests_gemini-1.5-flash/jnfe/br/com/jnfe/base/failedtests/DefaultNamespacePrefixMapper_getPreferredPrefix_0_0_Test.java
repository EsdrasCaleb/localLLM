package br.com.jnfe.base;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import com.sun.xml.bind.marshaller.NamespacePrefixMapper;

public class DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test {

    private static final Logger logger = LoggerFactory.getLogger(DefaultNamespacePrefixMapper_getPreferredPrefix_0_0_Test.class);

    @Test
    void testGetPreferredPrefix_nf() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "suggestion", true);
        assertEquals("nf", prefix);
    }

    @Test
    void testGetPreferredPrefix_null_true() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("someOtherNamespace", "suggestion", true);
        assertNull(prefix);
    }

    @Test
    void testGetPreferredPrefix_null_false() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("someOtherNamespace", "suggestion", false);
        assertNull(prefix);
    }

    @Test
    void testGetPreferredPrefix_null_namespace() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix(null, "suggestion", true);
        assertNull(prefix);
    }

    @Test
    void testGetPreferredPrefix_nf_false() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "suggestion", false);
        assertEquals("nf", prefix);
    }

    @Test
    void testGetPreferredPrefix_empty_namespace() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("", "suggestion", true);
        assertNull(prefix);
    }

    @Test
    void testGetPreferredPrefix_empty_suggestion() {
        DefaultNamespacePrefixMapper mapper = new DefaultNamespacePrefixMapper();
        String prefix = mapper.getPreferredPrefix("http://www.portalfiscal.inf.br/nfe", "", true);
        assertEquals("nf", prefix);
    }
}
