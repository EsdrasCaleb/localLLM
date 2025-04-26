package com.densebrain.rif.client;

import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;
import java.io.IOException;
import java.rmi.RemoteException;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
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
import com.densebrain.rif.client.service.RIFService;
import com.densebrain.rif.client.service.types.Invoke;
import com.densebrain.rif.client.service.types.InvokeResponse;
import com.densebrain.rif.util.ObjectUtility;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;

@ExtendWith(MockitoExtension.class)
class RIFInvoker_invoke_2_1_Test {

    @Mock
    private RIFManager manager;

    @Mock
    private RIFService service;

    @Mock
    private Class interfaceClazz;

    @InjectMocks
    private RIFInvoker rifInvoker;

    @BeforeEach
    void setUp() throws RemoteException {
        rifInvoker = new RIFInvoker(manager, interfaceClazz);
    }

    @Test
    void testInvoke() throws Exception {
        // Mock the necessary objects and methods
        Invoke invoke = mock(Invoke.class);
        InvokeResponse invokeResponse = mock(InvokeResponse.class);
        when(manager.getService()).thenReturn(service);
        when(service.invoke(invoke)).thenReturn(invokeResponse);
        // Invoke the method using reflection
        Method method = RIFInvoker.class.getDeclaredMethod("invoke", Invoke.class);
        method.setAccessible(true);
        Object result = method.invoke(rifInvoker, invoke);
        // Verify the result
        assertEquals(invokeResponse, result);
        verify(manager).getService();
        verify(service).invoke(invoke);
    }

    @Test
    void testInvokeSerializationException() throws RemoteException, IOException {
        String methodName = "testMethod";
        Object[] params = new Object[] { "param1", "param2" };
        when(manager.getService()).thenReturn(service);
        doThrow(new IOException()).when(service).invoke(any(Invoke.class));
        assertThrows(RemoteException.class, () -> rifInvoker.invoke(methodName, params));
        verify(manager).getService();
    }

    @Test
    void testInvokeDeserializationException() throws RemoteException, IOException {
        String methodName = "testMethod";
        Object[] params = new Object[] { "param1", "param2" };
        Invoke invoke = new Invoke();
        invoke.setClassName(interfaceClazz.getName());
        invoke.setMethodName(methodName);
        invoke.setSerializedParams(ObjectUtility.encodeBytes(ObjectUtility.serializeObject(params)));
        InvokeResponse invokeResponse = new InvokeResponse();
        String serializedResponse = "invalidBase64";
        invokeResponse.set_return(serializedResponse);
        when(manager.getService()).thenReturn(service);
        when(service.invoke(invoke)).thenReturn(invokeResponse);
        assertThrows(RemoteException.class, () -> rifInvoker.invoke(methodName, params));
        verify(manager).getService();
        verify(service).invoke(invoke);
    }
}
