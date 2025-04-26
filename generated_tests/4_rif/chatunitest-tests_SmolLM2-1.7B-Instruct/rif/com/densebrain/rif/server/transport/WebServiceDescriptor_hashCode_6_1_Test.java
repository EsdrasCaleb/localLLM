package com.densebrain.rif.server.transport;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.mockito.junit.MockitoJUnitRunner;
import java.lang.reflect.Method;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@RunWith(MockitoJUnitRunner.class)
public class WebServiceDescriptor_hashCode_6_1_Test {

    @Mock
    private WebServiceDescriptor webServiceDescriptor;

    @InjectMocks
    private WebServiceDescriptor webServiceDescriptorUnderTest;

    @Test
    public void testHashCode() {
        // Arrange
        Class serviceClazz = WebServiceDescriptor.class;
        String targetNamespace = "targetNamespace";
        String typesNamespace = "typesNamespace";
        webServiceDescriptorUnderTest.setServiceClazz(serviceClazz);
        webServiceDescriptorUnderTest.setTargetNamespace(targetNamespace);
        webServiceDescriptorUnderTest.setTypesNamespace(typesNamespace);
        // Act
        int hashCode = webServiceDescriptorUnderTest.hashCode();
        // Assert
        assertNotNull(hashCode);
    }
}
