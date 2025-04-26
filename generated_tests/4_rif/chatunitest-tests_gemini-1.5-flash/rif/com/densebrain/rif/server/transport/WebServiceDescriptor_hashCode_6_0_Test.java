package com.densebrain.rif.server.transport;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_hashCode_6_0_Test {

    @Test
    void testHashCodeNullServiceClass() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(null, "targetNamespace", "typesNamespace");
        assertEquals(1, descriptor.hashCode());
    }

    @Test
    void testHashCodeNonNullServiceClass() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "targetNamespace", "typesNamespace");
        int hashCode = descriptor.hashCode();
        // Ensure it's not the default value
        assertNotEquals(1, hashCode);
        // Verify that the hashcode is consistent for the same service class.
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "targetNamespace", "typesNamespace");
        assertEquals(hashCode, descriptor2.hashCode());
    }

    @Test
    void testHashCodeDifferentServiceClasses() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "targetNamespace", "typesNamespace");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(Integer.class, "targetNamespace", "typesNamespace");
        assertNotEquals(descriptor1.hashCode(), descriptor2.hashCode());
    }

    @Test
    void testHashCodeSameServiceClassDifferentNamespaces() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "targetNamespace1", "typesNamespace1");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "targetNamespace2", "typesNamespace2");
        // Hashcode should only depend on serviceClazz
        assertEquals(descriptor1.hashCode(), descriptor2.hashCode());
    }
}
