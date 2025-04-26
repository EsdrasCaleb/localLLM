package com.densebrain.rif.server.transport;

import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;

public class WebServiceDescriptor_hashCode_6_1_Test {

    @Test
    public void testHashCode() {
        WebServiceDescriptor descriptor1 = new WebServiceDescriptor(null, "targetNamespace", "typesNamespace");
        WebServiceDescriptor descriptor2 = new WebServiceDescriptor(null, "targetNamespace", "typesNamespace");
        assertEquals(descriptor1.hashCode(), descriptor2.hashCode());
        descriptor1.setTargetNamespace("newTargetNamespace");
        descriptor2.setTargetNamespace("newTargetNamespace");
        assertEquals(descriptor1.hashCode(), descriptor2.hashCode());
        descriptor1.setTypesNamespace("newTypesNamespace");
        descriptor2.setTypesNamespace("newTypesNamespace");
        assertEquals(descriptor1.hashCode(), descriptor2.hashCode());
    }
}
