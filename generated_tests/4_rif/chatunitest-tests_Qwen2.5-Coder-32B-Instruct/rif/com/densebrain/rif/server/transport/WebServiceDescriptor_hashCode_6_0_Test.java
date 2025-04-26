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

    private WebServiceDescriptor webServiceDescriptor;

    @Mock
    private Class<?> mockServiceClass;

    @BeforeEach
    public void setUp() {
        MockitoAnnotations.openMocks(this);
        webServiceDescriptor = new WebServiceDescriptor(mockServiceClass, "http://example.com/target", "http://example.com/types");
    }

    @Test
    public void testHashCode() {
        when(mockServiceClass.hashCode()).thenReturn(12345);
        int expectedHashCode = Objects.hash(mockServiceClass, "http://example.com/target", "http://example.com/types");
        int actualHashCode = webServiceDescriptor.hashCode();
        assertEquals(expectedHashCode, actualHashCode);
    }

    @Test
    public void testHashCode_ServiceClassNotNull() {
        when(mockServiceClass.hashCode()).thenReturn(12345);
        int expectedHashCode = Objects.hash(mockServiceClass, "http://example.com/target", "http://example.com/types");
        int actualHashCode = webServiceDescriptor.hashCode();
        assertEquals(expectedHashCode, actualHashCode);
    }

    @Test
    public void testHashCode_ServiceClassNull() throws Exception {
        // Using reflection to set serviceClazz to null
        java.lang.reflect.Field serviceClazzField = WebServiceDescriptor.class.getDeclaredField("serviceClazz");
        serviceClazzField.setAccessible(true);
        serviceClazzField.set(webServiceDescriptor, null);
        int expectedHashCode = Objects.hash(null, "http://example.com/target", "http://example.com/types");
        int actualHashCode = webServiceDescriptor.hashCode();
        assertEquals(expectedHashCode, actualHashCode);
    }
}
