package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_equals_7_0_Test {

    @Mock
    private WebServiceDescriptor webServiceDescriptor;

    @InjectMocks
    private WebServiceDescriptor serviceDescriptor;

    @Test
    public void testEquals() {
        // Arrange
        webServiceDescriptor.setServiceClazz(WebServiceDescriptor.class);
        webServiceDescriptor.setTargetNamespace("targetNamespace");
        webServiceDescriptor.setTypesNamespace("typesNamespace");
        // Act
        Object obj = webServiceDescriptor;
        // Assert
        assertEquals("targetNamespace", serviceDescriptor.getTargetNamespace());
        assertEquals("typesNamespace", serviceDescriptor.getTypesNamespace());
    }
}
