package com.densebrain.rif.server.transport;

import java.lang.reflect.Method;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

public class WebServiceDescriptor_hashCode_6_1_Test {

    @Test
    public void testHashCode_ServiceClazzNull() {
        WebServiceDescriptor webServiceDescriptor = new WebServiceDescriptor(null, "targetNamespace", "typesNamespace");
        assertEquals(-169, webServiceDescriptor.hashCode());
    }

    @Test
    public void testHashCode_ServiceClazzNotNull() {
        WebServiceDescriptor webServiceDescriptor = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        assertEquals(-169, webServiceDescriptor.hashCode());
    }

    @Test
    public void testHashCode_ServiceClazzSameObject() {
        WebServiceDescriptor webServiceDescriptor = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        WebServiceDescriptor webServiceDescriptor2 = webServiceDescriptor;
        assertEquals(webServiceDescriptor.hashCode(), webServiceDescriptor2.hashCode());
    }

    @Test
    public void testHashCode_ServiceClazzDifferentObjects() {
        WebServiceDescriptor webServiceDescriptor = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        WebServiceDescriptor webServiceDescriptor2 = new WebServiceDescriptor(Class.class, "targetNamespace", "typesNamespace");
        assertNotEquals(webServiceDescriptor.hashCode(), webServiceDescriptor2.hashCode());
    }
}
