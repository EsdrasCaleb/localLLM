package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

@ExtendWith(MockitoExtension.class)
public class WebServiceDescriptor_equals_7_4_Test {

    @Mock
    private Class<?> serviceClass;

    @Test
    public void testEqualsWithSameObject() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        assertTrue(descriptor1.equals(descriptor1));
    }

    @Test
    public void testEqualsWithDifferentObjects() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        assertFalse(descriptor1.equals(descriptor2));
    }

    @Test
    public void testEqualsWithNullServiceClass() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(null, "ns1", "tns1");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        assertFalse(descriptor1.equals(descriptor2));
    }

    @Test
    public void testEqualsWithDifferentServiceClasses() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(serviceClass, "ns1", "tns1");
        assertFalse(descriptor1.equals(descriptor2));
    }
}
