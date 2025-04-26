package com.densebrain.rif.server.transport;

import java.util.Objects;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

class WebServiceDescriptor_equals_7_0_Test {

    @Test
    void testEquals_sameObject() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        Assertions.assertTrue(descriptor.equals(descriptor));
    }

    @Test
    void testEquals_nullObject() {
        WebServiceDescriptor descriptor = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        Assertions.assertFalse(descriptor.equals(null));
    }

    @Test
    void testEquals_differentClass() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        Object other = new Object();
        Assertions.assertFalse(descriptor1.equals(other));
    }

    @Test
    void testEquals_sameClass() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(String.class, "testNS2", "testTypesNS2");
        Assertions.assertTrue(descriptor1.equals(descriptor2));
    }

    @Test
    void testEquals_differentClassInstance() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(Integer.class, "testNS2", "testTypesNS2");
        Assertions.assertFalse(descriptor1.equals(descriptor2));
    }

    @Test
    void testEquals_nullServiceClazz() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(null, "testNS", "testTypesNS");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(null, "testNS2", "testTypesNS2");
        Assertions.assertTrue(descriptor1.equals(descriptor2));
        WebServiceDescriptor descriptor3 = new WebServiceDescriptor(String.class, "testNS", "testTypesNS");
        Assertions.assertFalse(descriptor1.equals(descriptor3));
    }
}
