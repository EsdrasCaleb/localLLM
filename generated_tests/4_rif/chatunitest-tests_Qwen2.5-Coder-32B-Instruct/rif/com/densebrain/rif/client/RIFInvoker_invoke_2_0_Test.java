package com.densebrain.rif.client;

import java.io.IOException;
import java.lang.reflect.Field;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.rmi.RemoteException;
import org.apache.bcel.Constants;
import org.apache.bcel.generic.ArrayType;
import org.apache.bcel.generic.BasicType;
import org.apache.bcel.generic.ClassGen;
import org.apache.bcel.generic.ConstantPoolGen;
import org.apache.bcel.generic.FieldGen;
import org.apache.bcel.generic.InstructionConstants;
import org.apache.bcel.generic.InstructionFactory;
import org.apache.bcel.generic.InstructionList;
import org.apache.bcel.generic.MethodGen;
import org.apache.bcel.generic.ObjectType;
import org.apache.bcel.generic.Type;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.types.Invoke;
import com.densebrain.rif.client.service.types.InvokeResponse;
import com.densebrain.rif.util.ObjectUtility;

@ExtendWith(MockitoExtension.class)
public class RIFInvoker_invoke_2_0_Test {

    @Mock
    private RIFManager manager;

    @Mock
    private RIFService service;

    @Mock
    private InvokeResponse invokeResponse;

    @InjectMocks
    private RIFInvoker rifInvoker;

    @BeforeEach
    void setUp() throws Exception {
        Field managerField = RIFInvoker.class.getDeclaredField("manager");
        managerField.setAccessible(true);
        managerField.set(rifInvoker, manager);
        Field interfaceClazzField = RIFInvoker.class.getDeclaredField("interfaceClazz");
        interfaceClazzField.setAccessible(true);
        interfaceClazzField.set(rifInvoker, String.class);
        when(manager.getService()).thenReturn(service);
    }

    @Test
    void testInvoke_Success() throws Exception {
        // Arrange
        String methodName = "testMethod";
        Object[] params = new Object[] { "param1", "param2" };
        String serializedParams = "serializedParams";
        String serializedResponse = "serializedResponse";
        Object expectedResponse = "expectedResponse";
        when(ObjectUtility.encodeBytes(ObjectUtility.serializeObject(params))).thenReturn(serializedParams);
        when(service.invoke(any(Invoke.class))).thenReturn(invokeResponse);
        when(invokeResponse.get_return()).thenReturn(serializedResponse);
        when(ObjectUtility.deserializeObjectBase64Encoded(serializedResponse)).thenReturn(expectedResponse);
        // Act
        Object result = rifInvoker.invoke(methodName, params);
        // Assert
        assertEquals(expectedResponse, result);
    }

    @Test
    void testInvoke_SerializationException() throws Exception {
        // Arrange
        String methodName = "testMethod";
        Object[] params = new Object[] { "param1", "param2" };
        IOException ioException = new IOException("Serialization failed");
        when(ObjectUtility.encodeBytes(ObjectUtility.serializeObject(params))).thenThrow(ioException);
        // Act & Assert
        RemoteException remoteException = assertThrows(RemoteException.class, () -> rifInvoker.invoke(methodName, params));
        assertEquals("Unable to serialize parameters", remoteException.getMessage());
        assertEquals(ioException, remoteException.getCause());
    }

    @Test
    void testInvoke_DeserializationException() throws Exception {
        // Arrange
        String methodName = "testMethod";
        Object[] params = new Object[] { "param1", "param2" };
        String serializedParams = "serializedParams";
        String serializedResponse = "serializedResponse";
        IOException ioException = new IOException("Deserialization failed");
        when(ObjectUtility.encodeBytes(ObjectUtility.serializeObject(params))).thenReturn(serializedParams);
        when(service.invoke(any(Invoke.class))).thenReturn(invokeResponse);
        when(invokeResponse.get_return()).thenReturn(serializedResponse);
        when(ObjectUtility.deserializeObjectBase64Encoded(serializedResponse)).thenThrow(ioException);
        // Act & Assert
        RemoteException remoteException = assertThrows(RemoteException.class, () -> rifInvoker.invoke(methodName, params));
        assertEquals("Unable to deserialize return value: Deserialization failed", remoteException.getMessage());
        assertEquals(ioException, remoteException.getCause());
    }
}
