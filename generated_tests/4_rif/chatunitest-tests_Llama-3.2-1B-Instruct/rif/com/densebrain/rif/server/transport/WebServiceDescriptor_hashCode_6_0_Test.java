package com.densebrain.rif.server.transport;

import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_hashCode_6_0_Test {

    @Mock
    private WebServiceDescriptor focal;

    @InjectMocks
    private WebServiceDescriptor service;

    @Test
    public void testHashCode() {
        // Given
        String targetNamespace = "targetNamespace";
        String typesNamespace = "typesNamespace";
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }

    @Test
    public void testHashCodeWithNullServiceClazz() {
        // Given
        String targetNamespace = "targetNamespace";
        String typesNamespace = "typesNamespace";
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }

    @Test
    public void testHashCodeWithNullServiceClazzAndNullTargetNamespace() {
        // Given
        String targetNamespace = "targetNamespace";
        String typesNamespace = null;
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }

    @Test
    public void testHashCodeWithNullServiceClazzAndNullTypesNamespace() {
        // Given
        String targetNamespace = "targetNamespace";
        String typesNamespace = null;
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }

    @Test
    public void testHashCodeWithNullTypesNamespaceAndNullServiceClazz() {
        // Given
        String targetNamespace = "targetNamespace";
        String typesNamespace = null;
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }

    @Test
    public void testHashCodeWithNullTargetNamespaceAndNullServiceClazz() {
        // Given
        String targetNamespace = null;
        String typesNamespace = "typesNamespace";
        // When
        int hashCode = focal.hashCode();
        // Then
        assertEquals(service.hashCode(), hashCode);
    }
}
