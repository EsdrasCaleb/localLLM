package com.densebrain.rif.server.transport;

import java.rmi.RemoteException;
import org.apache.axis2.AxisFault;
import org.apache.axis2.context.ConfigurationContext;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.mockito.*;
import org.junit.jupiter.api.*;
import static org.mockito.Mockito.*;
import static org.junit.jupiter.api.Assertions.*;
import java.net.InetAddress;
import java.util.LinkedList;
import java.util.List;
import org.apache.axis2.addressing.EndpointReference;
import org.apache.axis2.context.ConfigurationContextFactory;
import org.apache.axis2.description.AxisService;
import org.apache.axis2.rpc.receivers.RPCMessageReceiver;
import org.apache.axis2.transport.http.turnup.SimpleHTTPServer;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

@ExtendWith(MockitoExtension.class)
public class WebServiceContainer_configureService_2_3_Test {

    @Mock
    private ConfigurationContext configurationContext;

    @InjectMocks
    private WebServiceContainer webServiceContainer;

    @BeforeEach
    public void setUp() throws AxisFault, RemoteException {
        webServiceContainer = new WebServiceContainer(configurationContext, "contextPath", "hostName", 8080);
    }

    @Test
    public void testConfigureService() throws RemoteException {
        Class<?> serviceClazz = String.class;
        String targetNamespace = "targetNamespace";
        String typesNamespace = "typesNamespace";
        webServiceContainer.configureService(serviceClazz, targetNamespace, typesNamespace);
        // Verify that configureService(WebServiceDescriptor) is called with the correct descriptor
        verify(webServiceContainer, times(1)).configureService(any(WebServiceDescriptor.class));
    }
}
