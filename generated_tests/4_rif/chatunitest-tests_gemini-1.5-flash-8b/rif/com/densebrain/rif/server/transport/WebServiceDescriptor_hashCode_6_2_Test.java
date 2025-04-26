package com.densebrain.rif.server.transport;

import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_hashCode_6_2_Test {

    @Test
    void hashCode_nullServiceClazz() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(null, "target", "types");
        int hashCode = descriptor.hashCode();
        assertEquals(1, hashCode);
    }

    @Test
    void hashCode_nonNullServiceClazz() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "target", "types");
        int hashCode = descriptor.hashCode();
        // Check for a valid hash code
        assertTrue(hashCode > 1);
    }

    @Test
    void hashCode_differentServiceClazz() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "target", "types");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(Integer.class, "target", "types");
        assertNotEquals(descriptor1.hashCode(), descriptor2.hashCode());
    }

    @Test
    void hashCode_sameServiceClazz() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "target", "types");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "target2", "types2");
        assertEquals(descriptor1.hashCode(), descriptor1.hashCode());
        // Important to test for different objects with same class
        assertNotEquals(descriptor1.hashCode(), descriptor2.hashCode());
    }
}
