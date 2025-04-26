package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_hashCode_6_1_Test {

    @Mock
    private Class serviceClazz;

    private WebServiceDescriptor webServiceDescriptor;

    @BeforeEach
    public void setUp() {
        webServiceDescriptor = new WebServiceDescriptor(serviceClazz, "targetNamespace", "typesNamespace");
    }

    @Test
    public void testHashCode() {
        int expectedHashCode = webServiceDescriptor.getServiceClazz().hashCode() + webServiceDescriptor.getTargetNamespace().hashCode() + webServiceDescriptor.getTypesNamespace().hashCode();
        int actualHashCode = webServiceDescriptor.hashCode();
        assertEquals(expectedHashCode, actualHashCode);
    }
}
