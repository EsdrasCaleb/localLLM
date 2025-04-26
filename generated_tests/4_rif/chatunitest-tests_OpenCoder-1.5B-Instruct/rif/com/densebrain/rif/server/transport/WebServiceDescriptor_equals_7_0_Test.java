package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class WebServiceDescriptor_equals_7_0_Test {

    @InjectMocks
    private WebServiceDescriptor webServiceDescriptor;

    @Test
    void equalsObjectTest() {
        // Arrange
        Object obj = new Object();
        // Act
        boolean result = webServiceDescriptor.equals(obj);
        // Assert
        assertFalse(result);
    }
}
