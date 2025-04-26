package com.densebrain.rif.server.transport;

import java.lang.reflect.Field;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_equals_7_0_Test {

    @Test
    void testEqualsReflexive() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertTrue(descriptor.equals(descriptor));
    }

    @Test
    void testEqualsSymmetric() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertTrue(descriptor1.equals(descriptor2) && descriptor2.equals(descriptor1));
    }

    @Test
    void testEqualsTransitive() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertTrue(descriptor1.equals(descriptor2) && descriptor2.equals(descriptor3) && descriptor1.equals(descriptor3));
    }

    @Test
    void testEqualsNull() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertFalse(descriptor.equals(null));
    }

    @Test
    void testEqualsDifferentClass() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertFalse(descriptor.equals(new Object()));
    }

    @Test
    void testEqualsDifferentServiceClass() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(Integer.class, "ns1", "ns2");
        assertFalse(descriptor1.equals(descriptor2));
    }

    @Test
    void testEqualsSameServiceClass() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "ns3", "ns4");
        assertTrue(descriptor1.equals(descriptor2));
    }

    @Test
    void testEqualsNullServiceClass() throws NoSuchFieldException, IllegalAccessException {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(null, "ns1", "ns2");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "ns1", "ns2");
        assertFalse(descriptor1.equals(descriptor2));
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(null, "ns1", "ns2");
        assertTrue(descriptor1.equals(descriptor3));
    }
}
