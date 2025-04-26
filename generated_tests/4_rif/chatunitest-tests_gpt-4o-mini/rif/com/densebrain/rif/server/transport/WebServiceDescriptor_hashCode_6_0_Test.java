package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_hashCode_6_0_Test {

    private WebServiceDescriptor descriptorWithClass;

    private WebServiceDescriptor descriptorWithNullClass;

    @BeforeEach
    public void setUp() {
        descriptorWithClass = new WebServiceDescriptor(String.class, "http://example.com", "http://example.com/types");
        descriptorWithNullClass = new WebServiceDescriptor(null, "http://example.com", "http://example.com/types");
    }

    @Test
    public void testHashCodeWithNonNullServiceClazz() {
        int expectedHashCode = 31 + String.class.hashCode();
        assertEquals(expectedHashCode, descriptorWithClass.hashCode());
    }

    @Test
    public void testHashCodeWithNullServiceClazz() {
        assertEquals(1, descriptorWithNullClass.hashCode());
    }
}
