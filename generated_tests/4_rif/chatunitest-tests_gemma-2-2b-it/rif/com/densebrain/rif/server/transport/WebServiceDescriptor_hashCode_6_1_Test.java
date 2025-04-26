package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_hashCode_6_1_Test {

    WebServiceDescriptor webServiceDescriptor = new WebServiceDescriptor(WebServiceDescriptor.class, "http://example.com", "http://example.org");

    @Test
    void hashCodeTest() {
        // Arrange
        // Act
        int hashCode = webServiceDescriptor.hashCode();
        // Assert
        // Assert that hashCode is equal to expected value
        assertEquals(webServiceDescriptor.hashCode(), hashCode);
    }
}
